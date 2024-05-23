#pragma once

class Vec3 {
        float x_;
        float y_;
        float z_;

    public:
        Vec3() {}
        Vec3(float x, float y, float z) : x_(x), y_(y), z_(z) {}

        float x() const { return x_; }
        float y() const { return y_; }
        float z() const { return z_; }

        float r() const { return x_; }
        float g() const { return y_; }
        float b() const { return z_; }

        Vec3 operator-() const { return Vec3(-x_, -y_, -z_); }
        Vec3 operator+(const Vec3& other) const { return Vec3(x_ + other.x_, y_ + other.y_, z_ + other.z_); }
        Vec3 operator-(const Vec3& other) const { return Vec3(x_ - other.x_, y_ - other.y_, z_ - other.z_); }
        Vec3 operator*(const Vec3& other) const { return Vec3(x_ * other.x_, y_ * other.y_, z_ * other.z_); }
        Vec3 operator/(const Vec3& other) const { return Vec3(x_ / other.x_, y_ / other.y_, z_ / other.z_); }
        Vec3 operator+(const float t) const { return Vec3(x_ + t, y_ + t, z_ + t); }
        Vec3 operator-(const float t) const { return Vec3(x_ - t, y_ - t, z_ - t); }
        Vec3 operator*(const float t) const { return Vec3(x_ * t, y_ * t, z_ * t); }
        Vec3 operator/(const float t) const { return Vec3(x_ / t, y_ / t, z_ / t); }
};

// dot product
inline float dot(const Vec3& u, const Vec3& v) {
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

//


